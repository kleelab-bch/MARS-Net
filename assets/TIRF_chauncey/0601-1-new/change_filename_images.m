%change the filename.
mypath = 'image_all_v2';
filename = dir('image_all_v2/*.png');
fileNames = {filename.name};

mynewpath = 'image_all';
for iFile = 0: length(fileNames)-1  %# Loop over the file names
    filename = [num2str(iFile, '%05d') '.png'];
    newName = [num2str(iFile) '.png'];%sprintf('%04d.nc', iFile);  %# Make the new name
    f = fullfile(mynewpath, newName);
    g = fullfile(mypath, filename);
    copyfile(g, f);        %# Rename the file
end