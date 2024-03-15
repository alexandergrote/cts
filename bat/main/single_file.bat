@echo off

REM example call: my\path\single_file.bat malware baseline xgb 0 null

echo --- model %3 and random_state: %4 and features: %5 ---
call bat\classification\%1__%2.bat %3 %4 %5