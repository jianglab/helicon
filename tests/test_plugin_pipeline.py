"""The pipeline machinery shared by proc3d and images2star.

Both engines used to carry their own copy of the registry introspection and the
namespace builder; they had drifted apart. One copy serves both.
"""

import pytest

from helicon.lib import images2star_engine, plugin_pipeline, proc3d_engine
from helicon.plugins import images2star as images2star_plugins
from helicon.plugins import proc3d as proc3d_plugins


class TestMetavarText:
    def test_a_tuple_metavar_is_joined(self):
        assert plugin_pipeline.metavar_text(("x", "y")) == "x y"

    def test_a_missing_metavar_is_empty(self):
        assert plugin_pipeline.metavar_text(None) == ""
        assert plugin_pipeline.metavar_text("v") == "v"


class TestBothEnginesShareTheSpecs:
    @pytest.mark.parametrize(
        "engine, plugins, option",
        [
            (proc3d_engine, proc3d_plugins, "flip_hand"),
            (images2star_engine, images2star_plugins, "setParm"),
        ],
    )
    def test_the_engine_specs_come_from_the_shared_builder(
        self, engine, plugins, option
    ):
        specs = engine.operation_specs()
        assert specs == plugin_pipeline.operation_specs(plugins)
        assert option in specs
        assert set(specs[option]) == {
            "dest",
            "option_string",
            "metavar",
            "type",
            "nargs",
            "choices",
            "default",
            "help",
            "append",
        }


class TestStackToNamespace:
    def _specs(self):
        def add_args(parser):
            parser.add_argument("--thing", action="append")

        return plugin_pipeline.operation_specs(
            type(
                "P", (), {"_plugins": {"thing": type("M", (), {"add_args": add_args})}}
            )()
        )

    def test_extra_attributes_reach_the_namespace(self):
        args = plugin_pipeline.stack_to_namespace(
            [],
            self._specs(),
            lambda p: None,
            lambda args, extra: None,
            seed="value",
        )
        assert args.seed == "value"

    def test_a_repeated_append_option_accumulates(self):
        args = plugin_pipeline.stack_to_namespace(
            [("thing", "a"), ("thing", "b")],
            self._specs(),
            lambda p: None,
            lambda args, extra: None,
        )
        assert args.thing == ["a", "b"]

    def test_an_unknown_option_is_refused(self):
        with pytest.raises(ValueError, match="unknown operation"):
            plugin_pipeline.stack_to_namespace(
                [("missing", "x")],
                self._specs(),
                lambda p: None,
                lambda args, extra: None,
            )

    def test_a_non_append_option_may_not_repeat(self):
        add_args = lambda p: p.add_argument("--thing")  # noqa: E731
        specs = plugin_pipeline.operation_specs(
            type(
                "P", (), {"_plugins": {"thing": type("M", (), {"add_args": add_args})}}
            )()
        )
        with pytest.raises(ValueError, match="only be applied once"):
            plugin_pipeline.stack_to_namespace(
                [("thing", "a"), ("thing", "b")], specs, add_args, lambda a, e: None
            )

    def test_the_seeded_defaults_can_consume_extra_keys(self):
        seen = {}

        def seed_defaults(args, extra):
            args.verbose = 0
            extra.pop("inputMapFile", None)
            extra.pop("outputMapFile", None)
            seen["extra"] = dict(extra)

        args = plugin_pipeline.stack_to_namespace(
            [("thing", "a")],
            self._specs(),
            lambda p: None,
            seed_defaults,
            inputMapFile="/data/in.mrc",
            outputMapFile="/data/out.mrc",
        )
        assert args.verbose == 0
        assert args.thing == ["a"]
        # the keys the seeding consumed are not also set as plain attributes
        assert not hasattr(args, "inputMapFile")
        assert not hasattr(args, "outputMapFile")
        assert seen["extra"] == {}


class TestDefaultSeeding:
    def test_proc3d_seeds_its_cli_defaults(self):
        args = proc3d_engine.stack_to_namespace(
            [], proc3d_engine.operation_specs(), inputMapFile="/a.mrc"
        )
        assert args.verbose == 0
        assert args.cpu == 1
        assert args.force == 0
        assert args.inputMapFile == "/a.mrc"
        assert args.outputMapFile is None

    def test_images2star_seeds_its_cli_defaults(self):
        args = images2star_engine.stack_to_namespace(
            [], images2star_engine.operation_specs(), input_imageFiles="/a.mrcs"
        )
        assert args.verbose == 0
        assert args.cpu == 1
        assert args.input_imageFiles == ["/a.mrcs"]
        assert args.output_starFile == ""


def test_the_shared_builder_skips_a_plugin_without_add_args():
    module = type("M", (), {})()
    plugins = type("P", (), {"_plugins": {"broken": module}})()
    assert plugin_pipeline.operation_specs(plugins) == {}


def test_a_plugin_whose_add_args_raises_is_skipped():
    def boom(parser):
        raise RuntimeError("bad plugin")

    module = type("M", (), {"add_args": boom})()
    plugins = type("P", (), {"_plugins": {"broken": module}})()
    assert plugin_pipeline.operation_specs(plugins) == {}


def test_the_namespace_holds_the_argparse_defaults():
    def add_args(parser):
        parser.add_argument("--thing", default="d")

    specs = plugin_pipeline.operation_specs(
        type("P", (), {"_plugins": {"thing": type("M", (), {"add_args": add_args})}})()
    )
    got = plugin_pipeline.stack_to_namespace([], specs, add_args, lambda a, e: None)
    assert got.thing == "d"
