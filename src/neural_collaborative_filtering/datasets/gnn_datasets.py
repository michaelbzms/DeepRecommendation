from neural_collaborative_filtering.content_providers import GraphContentProvider
from neural_collaborative_filtering.datasets.base import PointwiseDataset, RankingDataset
from neural_collaborative_filtering.models.base import GNN_NCF


class GraphPointwiseDataset(PointwiseDataset):
    def __init__(self, file: str, graph_content_provider: GraphContentProvider):
        super().__init__(file)
        self.gcp = graph_content_provider

    def __getitem__(self, item):
        # return (user node index in graph, item node index in graph, target)
        userID, itemID, target = super().__getitem__(item)
        # from user & item IDs get their node IDs in the graph
        userNodeID = self.gcp.get_user_nodeID(userID)
        itemNodeID = self.gcp.get_item_nodeID(itemID)
        return userNodeID, itemNodeID, target

    def get_graph(self, device):
        return self.gcp.get_graph().to(device)

    @staticmethod
    def do_forward(model: GNN_NCF, batch, device, graph, *args):
        """ #xpects samples of (userId, itemId, target) and a graph to pass on to the model. """
        # get the input matrices and the target
        userIds, itemIds, y_batch = batch
        # forward model
        out = model(graph.to(device), userIds.long().to(device), itemIds.long().to(device), device, *args)
        return out, y_batch


class GraphRankingDataset(RankingDataset):
    def __init__(self, file: str, graph_content_provider: GraphContentProvider):
        super().__init__(file)
        self.gcp = graph_content_provider

    def __getitem__(self, item):
        # return (user node index in graph, item 1 node index in graph, item 2 node index in graph)
        userID, item1ID, item2ID = super().__getitem__(item)
        # from user & item IDs get their node IDs in the graph
        userNodeID = self.gcp.get_user_nodeID(userID)
        item1NodeID = self.gcp.get_item_nodeID(item1ID)
        item2NodeID = self.gcp.get_item_nodeID(item2ID)
        return userNodeID, item1NodeID, item2NodeID

    def get_graph(self, device):
        return self.gcp.get_graph().to(device)

    @staticmethod
    def do_forward(model: GNN_NCF, batch, device, graph, *args):
        """ Expects samples of (userId, item1Id, item2Id) and a graph to pass on to the model. """
        # get the input matrices and the target
        userIds, item1Ids, item2Ids = batch
        # forward model
        out1 = model(graph.to(device), userIds.long().to(device), item1Ids.long().to(device), device, *args)
        out2 = model(graph.to(device), userIds.long().to(device), item2Ids.long().to(device), device, *args)
        return out1, out2
