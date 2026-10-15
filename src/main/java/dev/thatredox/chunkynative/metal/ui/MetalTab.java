package dev.thatredox.chunkynative.metal.ui;

import dev.thatredox.chunkynative.metal.BlockAcceleration;
import dev.thatredox.chunkynative.metal.MetalSettings;
import dev.thatredox.chunkynative.metal.renderer.MetalSceneLoader;
import javafx.application.Platform;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.ui.render.RenderControlsTab;

/** Settings of the Metal renderer: how blocks are found (octree or ray tracing hardware). */
public class MetalTab implements RenderControlsTab {
    private final VBox box;
    private Scene scene;

    public MetalTab(Scene scene) {
        this.scene = scene;

        box = new VBox(10.0);
        box.setPadding(new Insets(10.0));

        ChoiceBox<String> loop = new ChoiceBox<>();
        String wavefront = "Wavefront (small kernels per bounce, faster)";
        String singleKernel = "Single kernel";
        loop.getItems().addAll(wavefront, singleKernel);
        loop.setValue(MetalSettings.wavefront() ? wavefront : singleKernel);
        loop.valueProperty().addListener((observable, previous, value) -> {
            if (value == null || value.equals(previous)) return;
            MetalSettings.setWavefront(value.equals(wavefront));
            this.scene.refresh();
        });

        ChoiceBox<BlockAcceleration> mode = new ChoiceBox<>();
        mode.getItems().addAll(BlockAcceleration.values());
        mode.setValue(BlockAcceleration.current());
        mode.valueProperty().addListener((observable, previous, value) -> {
            if (value == null || value == previous) return;
            BlockAcceleration.set(value);
            this.scene.refresh();
        });

        Label help = new Label("Ray tracing hardware (Apple M3 and newer) finds blocks faster than the octree in "
                + "large worlds. Smaller boxes are faster and need more memory (octree leaves: about 4 GB for a "
                + "12,000-chunk world); when they would need more than half of the GPU's memory, the octree is used.");
        help.setWrapText(true);

        CheckBox water = new CheckBox("Also trace water with ray tracing hardware (wavefront only, experimental)");
        water.setSelected(MetalSettings.hardwareWater());
        water.selectedProperty().addListener((observable, previous, value) -> {
            MetalSettings.setHardwareWater(value);
            this.scene.refresh();
        });

        Label status = new Label(MetalSceneLoader.accelerationStatus());
        status.setWrapText(true);
        MetalSceneLoader.setStatusListener(text -> Platform.runLater(() -> status.setText(text)));

        box.getChildren().addAll(
                new Label("Render loop (Metal renderer)"), loop,
                new Label("Block tracing (Metal renderer and preview)"), mode, help, water, status);
    }

    @Override
    public void update(Scene scene) {
        this.scene = scene;
    }

    @Override
    public String getTabTitle() {
        return "Metal";
    }

    @Override
    public Node getTabContent() {
        return box;
    }
}
