function [pts] = get3dLocs(total)
%GET3DLOCS

pts = [];
count = 1;
[~,x,y,z] = size(total);
for i=1:x
    for j=1:y
        for k=1:z
            if total(1,i,j,k) == 1
                pts(count,:) = [i,j,k];
                count = count + 1;
            end
        end
    end
end

end

