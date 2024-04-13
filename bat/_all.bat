@echo off

REM my/path/multi_file.bat synthetic

for %%f in ("spm", "feat_select") do (

    start bat\multi_file.bat %1 %%~f

)

