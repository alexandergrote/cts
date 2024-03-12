@echo off


for %%f in ("baseline", "boruta", "mutinfo", "cts") do (

    start bat\main\single_file %1 %%f

)