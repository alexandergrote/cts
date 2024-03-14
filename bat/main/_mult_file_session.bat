@echo off

REM call this file with two arguments: e.g. /my/file.bat synthetic baseline

for %%m in ("logistic_regression", "xgb") do (

    for %%s in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) do (

        call bat\main\single_file.bat %1 %2 %%~m %%s 

    )
)

exit 0