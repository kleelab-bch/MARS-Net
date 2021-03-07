dataset = "0601-1-half"
dir = "//research.wpi.edu/leelab/Chauncey/vUnet_TIRF_final/DataSet_label/" + dataset + "/mask/"
saved_dir = "//research.wpi.edu/leelab/Chauncey/vUnet_TIRF_final/DataSet_label/" + dataset + "/mask_smoothed_bin/"
File.makeDirectory(saved_dir); 
listimage = getFileList(dir);
for (i = 0; i < listimage.length; i++) {
	print(listimage[i]);
	open(dir + listimage[i]);
    run("Smooth");
    run("8-bit");
    setAutoThreshold("Default");
    run("Make Binary");
    run("Erode");
    run("Dilate");
    run("Make Binary");
    saveAs("PNG", saved_dir + listimage[i]);
    run("Close");
}

