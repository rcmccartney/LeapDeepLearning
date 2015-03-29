function image = ImageMaker(filename, dim, sampleSize, linesToSkip)

% dim is size of produced images
%open the current file
d = fopen(filename);
%error check
if d == -1
    error('Data file cannot be open - ERROR!')
end

%move position to beginning of matrix, skipping the header lines
for i=1:linesToSkip,
    fgetl(d);
end;

%set up the 15 images to use (5 fingers x 3 planes)
p = zeros(dim,dim,15);
% the 15 images after being resized will be stored here
q = zeros(sampleSize,sampleSize,15); 
% loop through file grabbing the finger positions
line = fgetl(d);
while line ~= -1,
    out = regexp(line, ' *', 'split');
    for i=1:1,
        index = 3*(i-1) + 1;
        % grab each finger picture out of full array
        % decay it by .98 in order to get temporal changes
        p1 = 0.98*squeeze(p(:,:,index));
        p2 = 0.98*squeeze(p(:,:,index+1));
        p3 = 0.98*squeeze(p(:,:,index+2));
        % get this line of data for this finger
        mat = fingers(out, dim, i);
        mat = mat(~any(isnan(mat),2),:);
        % calculate the linear indices
        xy_indices = sub2ind(size(p1), mat(:,1), mat(:,2));    
        yz_indices = sub2ind(size(p2), mat(:,2), mat(:,3));
        xz_indices = sub2ind(size(p3), mat(:,1), mat(:,3));
        % indexing into the image and update pixels
        p1(xy_indices) = p1(xy_indices) + .2;
        p2(yz_indices) = p2(yz_indices) + .2;
        p3(xz_indices) = p3(xz_indices) + .2;
        % stick back into full array 
        p(:,:,index) = p1;
        p(:,:,index+1) = p2;
        p(:,:,index+2) = p3;
    end;
    % go to next line
    line = fgetl(d);
end;

for i=1:15,
    % crop to only the gesture itself
    temp = squeeze(p(:,:,i));
    [row, col] = find(temp);   
    temp = temp(min(row):max(row),min(col):max(col));
    if isempty(temp),
        q(:,:,i) = zeros(sampleSize, sampleSize);  
    else
        q(:,:,i) = imresize(temp, [sampleSize sampleSize]);
    end;
end;
      
image = [q(:,:,1) q(:,:,2) q(:,:,3);
         q(:,:,4) q(:,:,5) q(:,:,6);
         q(:,:,7) q(:,:,8) q(:,:,9);
         q(:,:,10) q(:,:,11) q(:,:,12);
         q(:,:,13) q(:,:,14) q(:,:,15)];

%close file
fclose(d);
%returns the images made