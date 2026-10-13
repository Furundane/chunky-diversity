package dev.thatredox.chunkynative.common.renderer;

import se.llbit.chunky.main.Chunky;
import se.llbit.chunky.renderer.DefaultRenderManager;
import se.llbit.chunky.renderer.Renderer;
import se.llbit.chunky.renderer.SnapshotControl;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.util.TaskTracker;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

/**
 * Render loop shared by the GPU backends. The device accumulates passes in its own buffer; this
 * loop runs passes, merges the device buffer into the scene's sample buffer a few times per
 * second (and for snapshots and the last pass), and talks to the render manager.
 */
public abstract class GpuPathTracingRenderer implements Renderer {

    private BooleanSupplier postRender = () -> true;

    /** Device resources for rendering one scene. */
    protected interface RenderJob extends AutoCloseable {
        /** Whether camera rays are generated on the CPU and should be regenerated (with new jitter) between passes. */
        boolean needsCameraGeneration();

        /** Generates camera rays and uploads them, holding {@code renderLock} while writing device memory. */
        void generateCamera(Lock renderLock);

        /**
         * Runs one pass and waits for it to finish. The device buffer becomes
         * {@code (buffer * bufferSpp + sample) / (bufferSpp + 1)}.
         */
        void renderPass(int randomSeed, int bufferSpp);

        /** Copies the device buffer (3 floats per pixel) into {@code passBuffer}. */
        void readBuffer(float[] passBuffer);

        @Override
        void close();
    }

    /**
     * Loads the scene if needed and creates the device resources for rendering it. The device
     * buffer starts out as {@code passBuffer} (all zero).
     */
    protected abstract RenderJob createJob(Scene scene, float[] passBuffer);

    @Override
    public void setPostRender(BooleanSupplier callback) {
        postRender = callback;
    }

    @Override
    public void render(DefaultRenderManager manager) throws InterruptedException {
        ReentrantLock renderLock = new ReentrantLock();
        Scene scene = manager.bufferedScene;

        double[] sampleBuffer = scene.getSampleBuffer();
        float[] passBuffer = new float[sampleBuffer.length];

        try (RenderJob job = createJob(scene, passBuffer)) {

            // Generate initial camera rays
            job.generateCamera(renderLock);

            // bufferSppReal: GPU passes accumulated in the device buffer but not yet merged into the sample buffer.
            // logicalSpp: samples merged (or being merged) into the sample buffer. scene.spp is only advanced once a
            // merge has finished, so the render manager never sees samples that are not in the sample buffer yet.
            int bufferSppReal = 0;
            int logicalSpp = scene.spp;
            final int[] sceneSpp = {scene.spp};
            long lastCallback = 0;
            long lastMerge = System.currentTimeMillis();

            // Seed from the current spp so a resumed render does not repeat the samples it already has.
            Random rand = new Random(scene.spp);

            ForkJoinTask<?> cameraGenTask = Chunky.getCommonThreads().submit(() -> 0);
            ForkJoinTask<?> bufferMergeTask = Chunky.getCommonThreads().submit(() -> 0);

            // This is the main rendering loop. This deals with dispatching rendering tasks. The majority of time is spent
            // waiting for the GPU to complete.
            while (logicalSpp < scene.getTargetSpp()) {
                renderLock.lock();
                try {
                    job.renderPass(rand.nextInt(), bufferSppReal);
                } finally {
                    renderLock.unlock();
                }
                bufferSppReal += 1;

                if (job.needsCameraGeneration() && cameraGenTask.isDone()) {
                    cameraGenTask = Chunky.getCommonThreads().submit(() -> job.generateCamera(renderLock));
                }

                int reachedSpp = logicalSpp + bufferSppReal;
                boolean lastPass = reachedSpp >= scene.getTargetSpp();
                boolean forceMerge = lastPass || isSaveEvent(manager.getSnapshotControl(), scene, reachedSpp);
                long time = System.currentTimeMillis();

                if (!forceMerge) {
                    if (!bufferMergeTask.isDone()) {
                        continue;
                    }
                    // Merging and post processing the full frame costs CPU time on every core, so limit it to a
                    // few times per second while the canvas is visible, and once per second otherwise.
                    long mergeInterval = scene.shouldFinalizeBuffer() ? 250 : 1000;
                    if (bufferSppReal < 1024 && time - lastMerge < mergeInterval) {
                        // Not merging yet; still give the render manager a chance to pause or reset.
                        if (time - lastCallback > 100 && !manager.shouldFinalize()) {
                            lastCallback = time;
                            if (postRender.getAsBoolean()) break;
                        }
                        continue;
                    }
                }

                // Merge the accumulated GPU passes into the sample buffer.
                bufferMergeTask.join();
                job.readBuffer(passBuffer);
                int sampSpp = sceneSpp[0];
                int passSpp = bufferSppReal;
                double sinv = 1.0 / (sampSpp + passSpp);
                bufferSppReal = 0;
                lastMerge = time;

                bufferMergeTask = Chunky.getCommonThreads().submit(() -> {
                    Arrays.parallelSetAll(sampleBuffer, i -> (sampleBuffer[i] * sampSpp + passBuffer[i] * passSpp) * sinv);
                    sceneSpp[0] += passSpp;
                    scene.spp = sceneSpp[0];
                    scene.postProcessFrame(TaskTracker.Task.NONE);
                    manager.redrawScreen();
                });
                logicalSpp += passSpp;

                if (forceMerge) {
                    // Snapshots and the end of the render need the merged samples.
                    bufferMergeTask.join();
                }
                if (forceMerge || time - lastCallback > 100) {
                    lastCallback = time;
                    if (postRender.getAsBoolean()) break;
                }
            }

            cameraGenTask.join();
            bufferMergeTask.join();
        }
    }

    private boolean isSaveEvent(SnapshotControl control, Scene scene, int spp) {
        return control.saveSnapshot(scene, spp) || control.saveRenderDump(scene, spp);
    }

    @Override
    public boolean autoPostProcess() {
        return false;
    }
}
