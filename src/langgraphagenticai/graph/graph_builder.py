from langgraph.graph import StateGraph
from src.langgraphagenticai.state.state import CallState
from src.langgraphagenticai.nodes.nodes import CallCenterNode
from langgraph.graph import START, END


class GraphBuilder:
    def __init__(self, model):
        self.llm = model
        self.graph_builder = StateGraph(CallState)
        self.nodes = CallCenterNode(llm=self.llm)

    def call_center_build_graph(self):
        self.graph_builder.add_node("preprocess_node", self.nodes.preprocess_node)
        self.graph_builder.add_node("nlu_node", self.nodes.nlu_node)

        self.graph_builder.add_node("billing_issue_node", self.nodes.billing_issue_node)
        self.graph_builder.add_node("sim_not_working_node", self.nodes.sim_not_working_node)
        self.graph_builder.add_node("no_network_coverage_node", self.nodes.no_network_coverage_node)
        self.graph_builder.add_node("internet_speed_slow_node", self.nodes.internet_speed_slow_node)
        self.graph_builder.add_node("data_not_working_after_recharge_node", self.nodes.data_not_working_after_recharge_node)
        self.graph_builder.add_node("call_drops_frequently_node", self.nodes.call_drops_frequently_node)

        self.graph_builder.add_edge(START, "preprocess_node")
        self.graph_builder.add_edge("preprocess_node", "nlu_node")

        self.graph_builder.add_conditional_edges("nlu_node", self.nodes.route_intent_to_node, { 
            "billing_issue_node": "billing_issue_node",
            "sim_not_working_node": "sim_not_working_node",
            "no_network_coverage_node": "no_network_coverage_node",
            "internet_speed_slow_node": "internet_speed_slow_node",
            "data_not_working_after_recharge_node": "data_not_working_after_recharge_node",
            "call_drops_frequently_node": "call_drops_frequently_node"
        })

        for node in ["billing_issue_node", "sim_not_working_node", "no_network_coverage_node", "internet_speed_slow_node", "data_not_working_after_recharge_node", "call_drops_frequently_node"]:
            self.graph_builder.add_edge(node, END)

    def setup_graph(self):
        return self.graph_builder.compile()