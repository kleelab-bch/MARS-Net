function image_bw=find_max(image_bw)
L = bwlabel(image_bw);
stats = regionprops(L);
Ar = cat(1, stats.Area);
ind = find(Ar == max(Ar));
image_bw(L~=ind)=0;
end
