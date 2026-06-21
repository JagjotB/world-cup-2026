@echo off
REM Wrapper for the World Cup 2026 auto-updater, used by the scheduled task.
REM Runs one update cycle (refresh schedule, regenerate predictions, sync
REM results log) and commits + pushes any new results.
cd /d "%~dp0.."
python scripts\auto_update.py --commit %*
