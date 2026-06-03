function [Countrate0, Countrate1]=GetPL(device)
Countrate0 = 0;
Countrate1 = 0;
CountratePtr = libpointer('int32Ptr', Countrate0);
[ret, Countrate0] = calllib('PHlib', 'PH_GetCountRate', device,0,CountratePtr);
CountratePtr = libpointer('int32Ptr', Countrate1);
[ret, Countrate1] = calllib('PHlib', 'PH_GetCountRate', device,1,CountratePtr);

%fprintf('\nCountrate0=%1d/s Countrate1=%1d/s', Countrate0, Countrate1);