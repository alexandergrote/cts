@echo off

REM call this file with two arguments: e.g. /my/file.bat synthetic baseline

for %%m in ("logistic_regression", "xgb", "svm") do (

    for %%s in (0, 1) do (

        for %%f in (null, 20, 40, 60, 80) do (

            call bat\single_file.bat %1 %2 %%~m %%s %%f %3

        )
    )
)

exit 0