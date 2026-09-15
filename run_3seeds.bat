@echo off
setlocal
set PY=C:\Python314\python.exe
set REPO=C:\Users\USER\Documents\Project\GMN-Football-3-main
set LOGDIR=%REPO%\training\results\forensics
set SCENARIO=academy_3_vs_1_with_keeper
set TIMESTEPS=200000
set BRPORT=5050
cd /d %REPO%
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
echo [orchestrator] started at %DATE% %TIME% > "%LOGDIR%\_orch.log" 2>&1

:waitport
netstat -ano | findstr ":%BRPORT% " | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% equ 0 (
    timeout /t 30 /nobreak >nul 2>&1
    goto waitport
)

echo [orchestrator] port %BRPORT% free, starting seed 123 >> "%LOGDIR%\_orch.log" 2>&1
%PY% -m training.train_mappo --scenario %SCENARIO% --seed 123 --timesteps %TIMESTEPS% --n-envs 1 --checkpoint-name mappo_%SCENARIO%_seed123_quarantine.pt > "%LOGDIR%\train_seed123.log" 2>&1
echo [orchestrator] seed 123 done at %DATE% %TIME% >> "%LOGDIR%\_orch.log" 2>&1

:waitport2
netstat -ano | findstr ":%BRPORT% " | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% equ 0 (
    timeout /t 30 /nobreak >nul 2>&1
    goto waitport2
)

echo [orchestrator] port free, starting seed 999 >> "%LOGDIR%\_orch.log" 2>&1
%PY% -m training.train_mappo --scenario %SCENARIO% --seed 999 --timesteps %TIMESTEPS% --n-envs 1 --checkpoint-name mappo_%SCENARIO%_seed999_quarantine.pt > "%LOGDIR%\train_seed999.log" 2>&1
echo [orchestrator] seed 999 done at %DATE% %TIME% >> "%LOGDIR%\_orch.log" 2>&1

echo [orchestrator] all seeds trained, starting evals >> "%LOGDIR%\_orch.log" 2>&1
for %%S in (42 123 999) do (
    call :eval_seed %%S
)

:waitportfinal
netstat -ano | findstr ":%BRPORT% " | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% equ 0 (
    timeout /t 30 /nobreak >nul 2>&1
    goto waitportfinal
)
%PY% -m pytest training/tests/test_reward_exploits.py -q > "%LOGDIR%\exploit_suite.log" 2>&1
echo [orchestrator] exploit suite done, exit=%ERRORLEVEL% >> "%LOGDIR%\_orch.log" 2>&1
echo [orchestrator] ALL DONE at %DATE% %TIME% >> "%LOGDIR%\_orch.log" 2>&1
goto :eof

:eval_seed
:weval
netstat -ano | findstr ":%BRPORT% " | findstr LISTENING >nul 2>&1
if %ERRORLEVEL% equ 0 (
    timeout /t 30 /nobreak >nul 2>&1
    goto weval
)
set BEST=training/models/mappo_%SCENARIO%_seed%1_best.pt
if not exist "%BEST%" (
    echo [orchestrator] WARNING: %BEST% not found, skipping >> "%LOGDIR%\_orch.log" 2>&1
    goto :eof
)
%PY% training/eval_mappo.py --checkpoint "%BEST%" --episodes 50 --deterministic > "%LOGDIR%\eval_seed%1.log" 2>&1
echo [orchestrator] eval seed %1 done >> "%LOGDIR%\_orch.log" 2>&1
goto :eof

