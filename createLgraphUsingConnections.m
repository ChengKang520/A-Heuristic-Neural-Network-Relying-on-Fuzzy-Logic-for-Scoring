function lgraph = createLgraphUsingConnections(layers,connections)
% lgraph = createLgraphUsingConnections(layers,connections) creates a layer
% graph with the layers in the layer array |layers| connected by the
% connections in |connections|.

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end

function c = subgraphConnections(src,dest)
if numel(src) > 1
    dest_name_root = [dest.Name '/in'];
    tmp = cell(numel(src),2);
    for k = 1:numel(src)
        tmp{k,1} = src(k).Name;
        tmp{k,2} = [dest_name_root num2str(k)];
    end
elseif numel(dest) > 1
    tmp = cell(numel(dest),2);
    for k = 1:numel(dest)
        tmp{k,1} = src.Name;
        tmp{k,2} = dest(k).Name;
    end
end
c = table(tmp(:,1),tmp(:,2),'VariableNames',{'Source','Destination'});
end