from shiny import App, ui, reactive, render
import random

# Define the UI
app_ui = ui.page_fluid(
    ui.h2("Dynamic UI with Python Shiny"),
    ui.input_action_button("add_button", "Add Input"),
    ui.div(id="dynamic_inputs"),
    ui.div(id="results", style="margin-top: 20px;"),
)

# Define the server logic
def server(input, output, session):
    cnt = reactive.Value(0)
    # A reactive value to keep track of input IDs
    dynamic_inputs = reactive.Value([])

    # Observer to handle the "Add Input" button click
    @reactive.Effect
    @reactive.event(input.add_button)
    def _():
        current_inputs = dynamic_inputs.get()
        print("current inputs:", current_inputs)
        new_input_id = f"input_{len(current_inputs) + 1}"
        dynamic_inputs.set(current_inputs + [new_input_id])

        # Dynamically insert a new numeric input
        ui.insert_ui(
            ui.input_numeric(new_input_id, f"Input {len(current_inputs) + 1}", value=0),
            selector="#dynamic_inputs"
        )
        @reactive.Effect
        @reactive.event(input[new_input_id],ignore_init=True)
        def _():
            print(input[new_input_id]()**2)        
        
        cnt.set(dynamic_inputs())
        print(cnt())
        print(input[new_input_id])

    # # Reactive functions to handle each input dynamically
    # @reactive.Effect
    # @reactive.event(cnt)
    # def compute_results():
        # print([input[f"input_{i+1}"] for i in range(len(dynamic_inputs()))])
        
        # # each time this runs, it will assign a new listener function to input_1, but earlier ones still preserve. Need use jquery's .off() method to remove previous listener(?)
        # for i, input_ui in enumerate([input[f"input_{i+1}"] for i in range(len(dynamic_inputs()))]):
            # @reactive.Effect
            # @reactive.event(input[f"input_{i+1}"],ignore_init=True)
            # def _():
                # print(input[f"input_{i+1}"]())
            # @reactive.Effect
            # @reactive.event(input[f"input_{i+1}"],ignore_init=True)
            # def _():
                # print(input[f"input_{i+1}"]()**2)
            # break
            

    # Output the results dynamically
    @output
    @render.ui
    def results():
        return ui.HTML(compute_results())

# Run the app
app = App(app_ui, server)
