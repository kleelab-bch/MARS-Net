%change the filename.
mypath = 'mask_smoothed_bin_v2';
filename = dir('mask_smoothed_bin_v2/*.tif');
fileNames = {filename.name};
index = 0;
for iFile = 1: length(fileNames)  %# Loop over the file names
      newName = [num2str(index) '-m.tif'];%sprintf('%04d.nc', iFile);  %# Make the new name
      f = fullfile(mypath, newName);
      g = fullfile(mypath, fileNames{iFile});
      movefile(g,f);        %# Rename the file
      index = index + 5;
end