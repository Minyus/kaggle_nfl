from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from .sub_pipeline import SubPipeline
from typing import Callable, Union, List, Iterable  # NOQA


class KedexPipeline(Pipeline):
    def __init__(
        self,
        nodes,  # type: Iterable[Union[Node, "Pipeline", "KedroPipeline"]]
        *,
        parameters_in_inputs=False,  # type: bool
        main_input_index=0,  # type: int
        module="",  # type: str
        decorator=[],  # type: Union[Callable, List[Callable]]
        name=None  # type: str
    ):

        for i, node in enumerate(nodes):
            if isinstance(node, dict):

                if parameters_in_inputs:
                    inputs = node.get("inputs")
                    inputs = inputs if isinstance(inputs, list) else [inputs]
                    if not ("parameters" in inputs):
                        node["inputs"] = inputs + ["parameters"]

                # if callable(node.get("func")):
                #     nodes[i] = Node(**node)
                # else:
                if 1:
                    node.setdefault("main_input_index", main_input_index)
                    node.setdefault("module", module)
                    node.setdefault("decorator", decorator)
                    nodes[i] = SubPipeline(**node)

        super().__init__(nodes=nodes, name=name)
