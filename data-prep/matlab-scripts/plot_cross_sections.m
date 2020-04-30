function [] = plot_cross_sections(grid, x, y, z)

% x
% y
% z

figure();

xx = [];
yy = [];
zz = [];

cross_sec = grid(x,:,:);
cross_sec = reshape(cross_sec, [120,120]);
for i=1:size(cross_sec,1)
    for j=1:size(cross_sec,2)
       if cross_sec(i,j) == 1
          xx = [xx, i];
          yy = [yy, j];
       end
    end
end

subplot(3,1,1);
scatter(xx,yy);
title("Min X: " + min(xx) + ", Max X: " + max(xx) + ", Min Y: " + min(yy) + ", Max Y: " + max(yy));
xlim([0 120]);
ylim([0 120]);

xx = [];
yy = [];
zz = [];

cross_sec = grid(:,y,:);
cross_sec = reshape(cross_sec, [120,120]);
for i=1:size(cross_sec,1)
    for j=1:size(cross_sec,2)
       if cross_sec(i,j) == 1
          xx = [xx, i];
          yy = [yy, j];
       end
    end
end

subplot(3,1,2);
scatter(xx,yy);
title("Min X: " + min(xx) + ", Max X: " + max(xx) + ", Min Y: " + min(yy) + ", Max Y: " + max(yy));
xlim([0 120]);
ylim([0 120]);

xx = [];
yy = [];
zz = [];

cross_sec = grid(:,:,z);
cross_sec = reshape(cross_sec, [120,120]);
for i=1:size(cross_sec,1)
    for j=1:size(cross_sec,2)
        if cross_sec(i,j) == 1
            xx = [xx, i];
            yy = [yy, j];
        end
    end
end

subplot(3,1,3);
scatter(xx,yy);
title("Min X: " + min(xx) + ", Max X: " + max(xx) + ", Min Y: " + min(yy) + ", Max Y: " + max(yy));
xlim([0 120]);
ylim([0 120]);

end

