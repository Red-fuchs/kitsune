#!/bin/zsh
cd ${0:A:h} && swiftc -O main.swift -o kitsune-aec 2>&1 | grep -v "deprecated\|^ *[0-9]* |\|^ *|\|^$\|DeprecatedDeclaration"; ./kitsune-aec --self-check | grep -c "^ok"
