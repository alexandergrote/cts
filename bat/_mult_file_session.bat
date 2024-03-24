@echo off

REM call this file with two arguments: e.g. /my/file.bat synthetic baseline

if "%1" == "synthetic" (
    set features=null, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
)

if "%1" == "malware" (
    set features=null, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
)

if "%1" == "churn" (
    set features=null, 1, 2, 3, 4, 5
)

for %%m in ("logistic_regression", "xgb", "nb") do (

    for %%s in (0, 1) do (

        for %%f in (%features%) do (

            call bat\single_file.bat %1 %2 %%~m %%s %%f %3

        )
    )
)

exit 0