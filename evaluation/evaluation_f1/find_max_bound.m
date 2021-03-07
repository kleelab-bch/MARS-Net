function index=find_max_bound(B)
max = 0;
index = 1;
s_B = size(B,1);
if s_B > 1
    for s_index = 1:s_B
       b_temp = size(B{s_index},1);
       if b_temp > max
           max = b_temp;
           index = s_index;
       end
    end
end
end
