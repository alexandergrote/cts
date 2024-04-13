@echo off

REM my/path/multi_file.bat synthetic

for %%f in ("mutinfo", "rf", "cts") do (

    start bat\_mult_file_session.bat %1 %%~f %2

)