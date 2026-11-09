// kitsune-aec: Apple Voice Processing (AEC) audio helper. playback in (24 kHz f32) -> speaker; echo-cancelled mic (16 kHz f32) out.
// Playback and mic share ONE voice-processing unit (the echo reference is what we play through it).
// Protocol (little endian, frame = type:u8 len:u32 payload), no GUI, no bundle.
//   stdin : 'P' f32@24k playback | 'B' u32 pre-samples (new utterance: reset) | 'F' finish (play the rest even below pre-buffer) | 'S' stop+flush | 'U' pause | 'R' resume | 'Q' quit
//   stdout: 'H' json hello | 'M' f32@16k mic | 'T' u64 consumed, u64 queued, u32 underruns | 'E' text
// Modes: default = real audio (opens mic+speaker!); --null = same protocol, no audio device (mic = silence, playback drained at real time); --self-check = offline unit checks; --probe = create engine + enable VP, print formats, do not start.
import Foundation
import AVFoundation

setvbuf(stdout, nil, _IOFBF, 1 << 16)
let outLock = NSLock()
func send(_ t: UInt8, _ p: Data = Data()) {
    outLock.lock(); defer { outLock.unlock() }
    var h = Data([t]); var n = UInt32(p.count).littleEndian
    withUnsafeBytes(of: &n) { h.append(contentsOf: $0) }
    let d = h + p; _ = d.withUnsafeBytes { fwrite($0.baseAddress, 1, d.count, stdout) }; fflush(stdout)
}
func f32data(_ a: [Float]) -> Data { a.withUnsafeBufferPointer { Data(buffer: $0) } }
func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

final class PlayQ {
    private let l = NSLock(); private var b = [Float](); private var h = 0
    var pre = 6000, playing = false, finishing = false, paused = false
    var consumed: UInt64 = 0, under: UInt32 = 0
    func reset(pre: Int) { l.lock(); b = []; h = 0; self.pre = pre; playing = false; finishing = false; paused = false; consumed = 0; under = 0; l.unlock() }
    func push(_ x: [Float]) { l.lock(); b.append(contentsOf: x); l.unlock() }
    func flush() { l.lock(); b = []; h = 0; playing = false; finishing = false; paused = false; l.unlock() }
    func set(finish: Bool? = nil, paused p: Bool? = nil) { l.lock(); if let f = finish { finishing = f }; if let p = p { paused = p }; l.unlock() }
    var queued: Int { l.lock(); defer { l.unlock() }; return b.count - h }
    func snap() -> (UInt64, UInt64, UInt32) { l.lock(); defer { l.unlock() }; return (consumed, UInt64(b.count - h), under) }
    /// fills `out` (frames); silence while pre-buffering / paused / empty
    func pull(_ out: UnsafeMutablePointer<Float>, _ frames: Int) {
        l.lock(); defer { l.unlock() }
        if !playing && (b.count - h >= pre || (finishing && b.count - h > 0)) { playing = true }
        var n = 0
        if playing && !paused { n = min(frames, b.count - h); for i in 0..<n { out[i] = b[h + i] }; h += n; consumed += UInt64(n)
            if n < frames && !finishing { under += 1 }
            if h > 1 << 16 { b.removeFirst(h); h = 0 } }
        for i in n..<frames { out[i] = 0 }
    }
}
let q = PlayQ()

/// mono f32 at `inRate` -> 16 kHz blocks via AVAudioConverter
final class Down {
    let conv: AVAudioConverter; let inF: AVAudioFormat; let outF = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    init?(_ rate: Double) {
        guard let f = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false), let c = AVAudioConverter(from: f, to: outF) else { return nil }
        inF = f; conv = c
    }
    func run(_ x: UnsafePointer<Float>, _ n: Int) -> [Float] {
        guard let ib = AVAudioPCMBuffer(pcmFormat: inF, frameCapacity: AVAudioFrameCount(n)) else { return [] }
        ib.frameLength = AVAudioFrameCount(n); ib.floatChannelData![0].update(from: x, count: n)
        let cap = AVAudioFrameCount(Double(n) * 16000 / inF.sampleRate) + 32
        guard let ob = AVAudioPCMBuffer(pcmFormat: outF, frameCapacity: cap) else { return [] }
        var fed = false; var err: NSError?
        conv.convert(to: ob, error: &err) { _, st in if fed { st.pointee = .noDataNow; return nil }; fed = true; st.pointee = .haveData; return ib }
        return Array(UnsafeBufferPointer(start: ob.floatChannelData![0], count: Int(ob.frameLength)))
    }
}

func readN(_ n: Int) -> Data? {
    var d = Data(count: n); var got = 0
    while got < n {
        let r = d.withUnsafeMutableBytes { read(0, $0.baseAddress! + got, n - got) }
        if r <= 0 { return nil }; got += r
    }
    return d
}

let args = CommandLine.arguments
if args.contains("--self-check") {
    var bad = 0
    func chk(_ c: Bool, _ m: String) { if !c { bad += 1; print("FAIL", m) } else { print("ok  ", m) } }
    // 1 queue: pre-buffer, pause, stop, finish
    let t = PlayQ(); t.reset(pre: 100); var o = [Float](repeating: 9, count: 64)
    t.push([Float](repeating: 1, count: 50)); o.withUnsafeMutableBufferPointer { t.pull($0.baseAddress!, 64) }
    chk(o.allSatisfy { $0 == 0 } && t.consumed == 0, "pre-buffer holds back playback")
    t.set(finish: true); o.withUnsafeMutableBufferPointer { t.pull($0.baseAddress!, 64) }
    chk(t.consumed == 50 && o[49] == 1 && o[50] == 0, "finish plays the short rest, pads silence")
    t.reset(pre: 10); t.push([Float](repeating: 2, count: 200)); t.set(paused: true); o.withUnsafeMutableBufferPointer { t.pull($0.baseAddress!, 64) }
    chk(t.consumed == 0 && o.allSatisfy { $0 == 0 }, "paused outputs silence, keeps queue")
    t.set(paused: false); o.withUnsafeMutableBufferPointer { t.pull($0.baseAddress!, 64) }; chk(t.consumed == 64 && t.queued == 136, "resume continues")
    t.flush(); chk(t.queued == 0, "stop flushes")
    // 2 resampler 48k -> 16k keeps a 440 Hz tone (energy ratio) and length
    if let d = Down(48000) {
        var tone = [Float](repeating: 0, count: 4800); for i in 0..<4800 { tone[i] = 0.5 * sin(2 * .pi * 440 * Float(i) / 48000) }
        var y = [Float](); var last = 0
        for _ in 0..<4 { let z = tone.withUnsafeBufferPointer { d.run($0.baseAddress!, 4800) }; last = z.count; y += z }   // first call carries the filter priming delay
        let rms = sqrt(y.dropFirst(100).map { $0 * $0 }.reduce(0, +) / Float(max(1, y.count - 100)))
        chk(abs(last - 1600) < 40 && y.count > 6000, "48k->16k steady block \(last) ~ 1600, total \(y.count)"); chk(abs(rms - 0.3536) < 0.04, "tone rms \(rms) ~ 0.354")
    } else { chk(false, "converter") }
    // 3 frame codec round trip
    send(0x4d, f32data([0.25, -0.5])); print("(frame written to stdout above)")
    exit(bad == 0 ? 0 : 1)
}

if args.contains("--probe") {
    let e = AVAudioEngine(); let i = e.inputNode
    do { try i.setVoiceProcessingEnabled(true); print("voice processing enabled:", i.isVoiceProcessingEnabled) } catch { print("VP enable failed:", error) }
    print("input fmt:", i.outputFormat(forBus: 0)); print("output fmt:", e.outputNode.inputFormat(forBus: 0))
    exit(0)
}

let nullMode = args.contains("--null")
var info: [String: Any] = ["mode": nullMode ? "null" : "vpio", "in_rate": 16000, "out_rate": 24000]
var engine: AVAudioEngine? = nil

if !nullMode {
    let e = AVAudioEngine(); engine = e; let inp = e.inputNode
    do { try inp.setVoiceProcessingEnabled(true) } catch { send(0x45, Data("enable voice processing failed: \(error)".utf8)); exit(2) }
    inp.isVoiceProcessingAGCEnabled = ProcessInfo.processInfo.environment["KITSUNE_AEC_AGC"] == "1"   // default: AGC off (stable levels for the VAD)
    if #available(macOS 14.0, *) { inp.voiceProcessingOtherAudioDuckingConfiguration = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min) }
    let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
    let src = AVAudioSourceNode(format: fmt) { _, _, frames, abl in
        let list = UnsafeMutableAudioBufferListPointer(abl)
        if let p = list[0].mData?.assumingMemoryBound(to: Float.self) { q.pull(p, Int(frames)) }
        return noErr
    }
    e.attach(src); e.connect(src, to: e.mainMixerNode, format: fmt)
    let inFmt = inp.outputFormat(forBus: 0); var down: Down? = nil; var pend = [Float]()
    info["vp_in_rate"] = inFmt.sampleRate; info["vp_in_channels"] = Int(inFmt.channelCount)
    let want = Int(ProcessInfo.processInfo.environment["KITSUNE_AEC_CH"] ?? "0") ?? 0   // the VP input is 9 ch on this Mac (probe); ch 0 = processed mic (unverified), others: see the per-channel RMS in logs/aec.log
    var ss = [Double](repeating: 0, count: 16), nn = 0
    inp.installTap(onBus: 0, bufferSize: 1024, format: nil) { buf, _ in
        guard buf.frameLength > 0, let ch = buf.floatChannelData else { return }
        if down == nil || abs(down!.inF.sampleRate - buf.format.sampleRate) > 1 { down = Down(buf.format.sampleRate) }
        for c in 0..<min(16, Int(buf.format.channelCount)) { var a = 0.0; for i in 0..<Int(buf.frameLength) { a += Double(ch[c][i] * ch[c][i]) }; ss[c] += a }
        nn += Int(buf.frameLength)
        if nn >= Int(buf.format.sampleRate) * 5 { log("mic rms per channel (5 s): " + (0..<min(16, Int(buf.format.channelCount))).map { String(format: "%.4f", (ss[$0] / Double(nn)).squareRoot()) }.joined(separator: " ")); ss = [Double](repeating: 0, count: 16); nn = 0 }
        let y = down?.run(ch[min(want, Int(buf.format.channelCount) - 1)], Int(buf.frameLength)) ?? []
        pend.append(contentsOf: y)
        while pend.count >= 800 { send(0x4d, f32data(Array(pend[0..<800]))); pend.removeFirst(800) }
    }
    e.prepare()
    do { try e.start() } catch { send(0x45, Data("engine start failed: \(error)".utf8)); exit(3) }
}
let js = try! JSONSerialization.data(withJSONObject: info); send(0x48, js)

// telemetry (+ null mode: drain playback at real time, mic = silence)
let tq = DispatchQueue(label: "tele")
var last = Date()
let timer = DispatchSource.makeTimerSource(queue: tq); timer.schedule(deadline: .now(), repeating: .milliseconds(20))
var nullMic = 0
timer.setEventHandler {
    if nullMode {
        let now = Date(); let dt = now.timeIntervalSince(last); last = now
        var tmp = [Float](repeating: 0, count: Int(dt * 24000)); tmp.withUnsafeMutableBufferPointer { q.pull($0.baseAddress!, $0.count) }
        nullMic += Int(dt * 16000); while nullMic >= 800 { send(0x4d, f32data([Float](repeating: 0, count: 800))); nullMic -= 800 }
    }
    let (c, qd, u) = q.snap(); var d = Data()
    var c1 = c.littleEndian, q1 = qd.littleEndian, u1 = u.littleEndian
    withUnsafeBytes(of: &c1) { d.append(contentsOf: $0) }; withUnsafeBytes(of: &q1) { d.append(contentsOf: $0) }; withUnsafeBytes(of: &u1) { d.append(contentsOf: $0) }
    send(0x54, d)
}
timer.resume()

// command loop (stdin); EOF = parent gone -> exit
while true {
    guard let h = readN(5) else { break }
    let t = h[0]; let n = Int(h.subdata(in: 1..<5).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
    let p = n > 0 ? (readN(n) ?? Data()) : Data()
    switch t {
    case 0x50: q.push(p.withUnsafeBytes { Array(UnsafeBufferPointer(start: $0.baseAddress!.assumingMemoryBound(to: Float.self), count: p.count / 4)) })
    case 0x42: q.reset(pre: Int(p.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian))
    case 0x46: q.set(finish: true)
    case 0x53: q.flush()
    case 0x55: q.set(paused: true)
    case 0x52: q.set(paused: false)
    case 0x51: engine?.stop(); exit(0)
    default: break
    }
}
engine?.stop()
