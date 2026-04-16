# Ham Radio Phone Patch Daemon

Small Python tool that listens to audio from SDRangel (via virtual cable) and triggers SIP calls based on a tone highet than 1250Hz.

If a tone above the configured frequency holds for a few seconds, it dials the number inside the file (you have to configure it).  
If already in a call, the same tone hangs up.

Built for experimenting with radio ↔ phone patch setups using MicroSIP.

No authentication or safeguards so any valid tone will trigger it, be careful with it!

Tested with VB-Audio Cable + SDRangel on Windows.
