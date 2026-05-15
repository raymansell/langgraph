from typing import TypedDict, Annotated

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

# schema of the state of our graph
# every node has access to this state dict
class MessageGraph(TypedDict):
    # with `add_messages` reducer:
    # old_state = {"messages": [msg1, msg2]}
    # new_update = {"messages": [msg3]}
    # result = {"messages": [msg1, msg2, msg3]}
    messages: Annotated[list[BaseMessage], add_messages]

# node names
REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    # returns the updated state dict
    # langgraph will use the `add_messages` reducer to append this list to the current state that holds all current existing messages
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    # res is an AIMessage, we cast it to a HumanMessage to trick the llm into thinking that
    # the critique/feedback of the tweet was made by a person
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# not an edge - this is a routing function for an edge
# it determines which node to go next
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={END:END,REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)


graph = builder.compile()
# print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()

def main():
    print("Hello LangGraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)


if __name__ == "__main__":
    main()
