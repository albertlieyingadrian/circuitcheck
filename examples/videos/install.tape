# Record from this folder (examples/videos) with the repo venv at the root: vhs validate.tape
Output install.mp4
Set Shell zsh
Set FontSize 20
Set Width 1280
Set Height 720
Set Padding 28
Set TypingSpeed 35ms
Set Theme {"name":"ember","background":"#15110e","foreground":"#e9e3db","cursor":"#e07a3f","selection":"#3a2419","black":"#15110e","red":"#e07a3f","green":"#8fbf8f","yellow":"#f2b48e","blue":"#6f9ff0","magenta":"#c9561c","cyan":"#a39a90","white":"#d9d2ca","brightBlack":"#6f675f","brightRed":"#f2b48e","brightGreen":"#a8d8a8","brightYellow":"#f8e6dc","brightBlue":"#9fbff5","brightMagenta":"#e07a3f","brightCyan":"#d9d2ca","brightWhite":"#ffffff"}
Hide
Type "cd ../.. && export PATH=$PWD/.venv/bin:$PATH PS1='%F{208}circuitcheck%f $ ' && setopt interactivecomments && clear"
Enter
Sleep 1s
Show
Type "# Install from a clone (PyPI: pip install circuitcheck, once released). No torch needed."
Enter
Sleep 1s
Type "uv venv -q /tmp/cc-venv && uv pip install -q --python /tmp/cc-venv/bin/python -e . && echo installed"
Enter
Sleep 12s
Type "/tmp/cc-venv/bin/circuitcheck --help"
Enter
Sleep 5s
Type "# Three commands: diff, validate, label (plus summarize). Every one runs offline on the bundled examples."
Enter
Sleep 4s
