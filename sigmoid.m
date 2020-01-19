function s = sigmoid(x,c)
s = 1./(1+exp(-c.*x));
end