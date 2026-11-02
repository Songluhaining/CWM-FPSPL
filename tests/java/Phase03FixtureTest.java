package phase03fixture;
import org.junit.Test;
import static org.junit.Assert.*;

/** Synthetic programs with real executions; never benchmark samples. */
public class Phase03FixtureTest {
    @Test public void assignmentChain() { assertEquals(8, Phase03Program.chain(3)); }
    @Test public void trueBranch() { assertEquals(6, Phase03Program.branch(2)); }
    @Test public void falseBranch() { assertEquals(2, Phase03Program.branch(-2)); }
    @Test public void loopWrites() { assertEquals(3, Phase03Program.loop(3)); }
    @Test public void recursiveCalls() { assertEquals(6, Phase03Program.recursive(3)); }
    @Test public void heapAlias() { assertEquals(7, new Phase03Program().alias()); }
    @Test public void assertionFailure() { assertEquals(0, Phase03Program.chain(3)); }
    @Test public void unsupportedArray() { assertEquals(1, Phase03Program.array()); }
    @Test public void missingStringValue() { String s = "bounded"; assertEquals(7, s.length()); }
    @Test public void optimizedLine() { assertEquals(4, Phase03Program.optimized(3)); }
}
