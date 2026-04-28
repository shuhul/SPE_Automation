pause(3);
if ~exist('instance1')
    Setup_LightField_Environment;
    instance1=lfm(true);
end
[int,wl]=instance1.acquire;