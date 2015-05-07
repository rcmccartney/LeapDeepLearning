% Author: Rob McCartney
% First visualizes the images being trained

wrong = [18 41 82 96 100 103 110 124 132 133 142 169 181 182 189 207 210 ...
    226 240 265 267 276 281 286 308 323 341 351 384 398 399 403 419 424 ...
    451 453 460 479 503 504 526 542 548 550 553 559 594 598 602 605 639 ...
    651 657 700 711 735 739 798 831 852 879 ];

predict = [ 5 6 3 3 5 11 5 7 4 1 7 4 9 11 9 9 11 8 8 ...
    7 4 7 4 10 1 7 6 8 5 5 1 10 7 4 8 8 5 9 10 5 8 8 ...
    11 11 8 8 3 8 8 5 7 11 8 7 11 9 1 9 4 5 10 ];

actual = [1 11 1 5 1 6 6 2 3 5 12 2 6 2 7 8 6 9 9 10 ...
    8 12 2 6 3 5 11 10 4 3 3 7 10 11 9 10 3 4 7 7 9  ...
    3 1 4 9 1 5 5 9 3 12 1 9 12 6 8 3 8 1 12 2 ];

addpath code
% need square images for the display program
resizing = [100 100];

input('Visualizing a subset of the data. Press Enter')
clf ;
data = load(fullfile('data','leap-binary-large', 'imdb.mat')) ;
% look at some random data
num = 20 ;
rows = [] ;
for i=1:num,
    rows = [rows ; reshape(imresize(data.images.data(:,:,:,i), resizing) ,1,[]) ] ;
end;
displayData(rows) ;

input('Training the classifier. Press Enter')
clf ; 
x = cnn_leap ;

input('Visualizing the first layer of filters. Press Enter')
clf ;
out = gather(x.layers{1, 1}.filters) ;
rows = [] ;
for i=1:size(out, 4),
    rows = [rows ; reshape(out(:,:,1,i).',1,[]) ] ;
end;
displayData(rows) ;

input('Visualizing the second layer of filters. Press Enter')
clf ;
out = gather(x.layers{1, 3}.filters) ;
rows = [] ;
for i=1:size(out, 4),
    rows = [rows ; reshape(out(:,:,1,i).',1,[]) ] ;
end;
displayData(rows) ;

input('Visualizing the errors. Press Enter')
clf ;
val = data.images.data(:,:,:,data.images.set==3);
rows = [] ;
for elm = wrong
    rows = [rows ; reshape(imresize(val(:,:,1,elm), resizing).',1,[]) ] ;
end;
displayData(rows) ;