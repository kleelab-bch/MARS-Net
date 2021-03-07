function [min_dist, Index] = calculate_mindist_withlocation(current_p, list_point, c_threshold)
%% 
%current_p:     the current point
%list_point:    the list of potential points in the other boundary.
%c_threshold:   maximum distance
%%
distances = pdist2(current_p, list_point);
[minDistance, Index] = min(distances);
min_dist = min(minDistance, c_threshold);
end