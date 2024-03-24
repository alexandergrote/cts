@echo off
setlocal enabledelayedexpansion


REM call this file with two arguments: e.g. /my/file.bat synthetic baseline 

if "%3"=="spm" (

    if %1==synthetic (
        set n_features_all=811
    )

    if %1==churn (
        set n_features_all=5
    )

    if %1==malware (
        set n_features_all=264
    )

) else (

    if %1==synthetic (
        set n_features_all=10
    )

    if %1==churn (
        set n_features_all=5
    )

    if %1==malware (
        set n_features_all=264
    )

)



for %%m in ("logistic_regression", "xgb", "svm") do (

    for %%s in (0, 1) do (

        for %%f in (null, 20, 40, 60, 80) do (

            if "%%f"=="null" (
                set /a n_features=n_features_all 
            ) else (
                set /a n_features=%%f * n_features_all / 100 
            )

            call bat\single_file_cts.bat %1 %2 %%~m %%s !n_features! %3

        )
    )
)


endlocal

exit 0