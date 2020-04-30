function [] = plot_occupancy(grid)

x = [];
y = [];
z = [];

for i=1:size(grid,1)
    for j=1:size(grid,2)
        for k=1:size(grid,3)
           if grid(i,j,k) == 1
              x = [x, i];
              y = [y, j];
              z = [z, k];
           end
        end
    end
end

% disp(max(x));
% disp(max(y));
% disp(max(z));

figure();
scatter3(x,y,z);

end

