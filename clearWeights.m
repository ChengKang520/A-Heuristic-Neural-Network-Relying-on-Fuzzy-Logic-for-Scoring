 function layers = clearWeights(layers)

for i = 1:numel(layers)
    if isprop(layers(i),'Weights')
        layers(i).Weights = zeros(size(layers(i).Weights));
    end
    if isprop(layers(i),'Bias')
        layers(i).Bias = zeros(size(layers(i).Bias));
    end
end