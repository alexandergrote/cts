@echo off

REM my/path/multi_file.bat synthetic

for %%s in ("synthetic", "malware", "churn") do (

    start /wait bat\_all.bat %%~s

)

