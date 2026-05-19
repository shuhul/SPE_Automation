"""
Emergency cleanup script. Run this if LightField hangs on startup.

What it does:
1. Tries to gracefully shutdown the bridge first (calls Dispose)
2. Kills any remaining Python processes running lf_bridge.py
3. After running this, LightField should start cleanly.

Usage:  python kill_bridge.py
"""
import socket, json, subprocess, sys, os, time

PORT = 27028

# 1. Try graceful shutdown first
print('Attempting graceful bridge shutdown...')
try:
    s = socket.socket()
    s.settimeout(2)
    s.connect(('127.0.0.1', PORT))
    s.sendall((json.dumps({'cmd': 'shutdown'}) + '\n').encode())
    s.recv(4096)
    s.close()
    print('  Bridge shut down gracefully.')
    time.sleep(1)
except Exception as e:
    print(f'  No bridge responded ({e}) — killing processes directly.')

# 2. Kill any Python processes running lf_bridge.py
try:
    result = subprocess.run(
        ['powershell', '-Command',
         'Get-WmiObject Win32_Process -Filter "name=\'python.exe\'" | '
         'Where-Object { $_.CommandLine -like \'*lf_bridge*\' } | '
         'ForEach-Object { Write-Host "Killing PID $($_.ProcessId)"; $_.Terminate() }'],
        capture_output=True, text=True
    )
    print(result.stdout or '  No lf_bridge processes found via WMI.')
except Exception as e:
    print(f'  WMI kill failed: {e}')

# 3. Fallback: kill by port (the process listening on 27028)
try:
    result = subprocess.run(
        ['powershell', '-Command',
         f'$pid = (Get-NetTCPConnection -LocalPort {PORT} -ErrorAction SilentlyContinue).OwningProcess; '
         f'if ($pid) {{ Write-Host "Killing PID $pid on port {PORT}"; Stop-Process -Id $pid -Force }} '
         f'else {{ Write-Host "No process on port {PORT}" }}'],
        capture_output=True, text=True
    )
    print(result.stdout.strip() or f'  No process on port {PORT}.')
except Exception as e:
    print(f'  Port kill failed: {e}')

print('\nDone. LightField should now start cleanly.')
