@echo off

REM my/path/multi_file.bat synthetic

for %%f in ("boruta", "mutinfo", "rf", "cts") do (

    if %%f=="cts" (
        echo start bat\_mult_file_session_cts.bat %1 %%~f %2
    ) else (

        start bat\_mult_file_session.bat %1 %%~f %2

    )   
)