%copy the image which are labelled.
% Get a list of all files and folders in this folder.
files = dir('./*');
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags);

for j = [ 7]%: length(subFolders)
    folder = subFolders(j, 1).name;
    filenames = dir([folder, '/mask/*.png']);
    source_folder = fullfile(folder, 'img_all');
    target_folder = fullfile(folder, 'img');
    mkdir(target_folder);

    for i = 1 : length(filenames)
        image = fullfile(source_folder, filenames(i, 1).name);
        copyfile(image, target_folder, 'f');
    end
end