function mat = fingers(out, dim, finger)
% for this data 42 starts the finger data and the x, y, z positions
% come every 7 out of 15 (so 48, 63, 78, etc)
%leap constants for converting coordinate system
leapStart = [-200, 0, -200];
leapRange = [400, 400, 400];

size = 1; %  need to change this for more rows of data
index = 48 + (finger-1)*15;  % this is where x,y,z data for finger 1-5 is in the file
mat = (cellfun(@str2double, [out(index), out(index+1), out(index+2)])-leapStart)*dim ./ leapRange;     

mat = round(mat);
%create pixels nearby these found points
mat = repmat(mat, 125, 1);
i = 1;
next = size - 1;
for x=-2:2,
    for y=-2:2
        for z=-2:2
            mat(i:i+next,:) = bsxfun(@plus,mat(i:i+next,:),[x y z]);
            i = i + next + 1;
        end
    end
end
% you could have created points less than 1, cap it at 1
% do the same for greater than dim, as that is out of bounds of the image

% invert y-axis to print correctly
mat(:,2) = dim - mat(:, 2) ; 
        
mat(mat < 1) = 1;
mat(mat > dim) = dim;