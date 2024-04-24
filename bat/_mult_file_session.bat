@echo off

REM call this file with two arguments: e.g. /my/file.bat synthetic baseline

set features=null, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

REM for %%m in ("logistic_regression", "xgb", "nb") do (
for %%m in ("logistic_regression", "nb") do (

    for %%s in (0, 1, 2, 3, 4) do (

        for %%f in (%features%) do (

            call bat\single_file.bat %1 %2 %%~m %%s %%f %3

        )
    )
)