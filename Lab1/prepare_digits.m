load digits.mat
position = [19 19];
radius = 6;
for i = 1:numel(digits_training)
    image = digits_training(i).image; 
    digits_training(i).descriptor = gradient_descriptor(image, position, radius); 
end
