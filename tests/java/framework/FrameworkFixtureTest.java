package frameworkfixture;
import org.junit.Test;
import static org.junit.Assert.*;
import static org.evosuite.runtime.EvoAssertions.verifyException;

public class FrameworkFixtureTest {
    @Test public void read() {
        try { FrameworkProgram.read(null); fail(); }
        catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
    }
    @Test public void write() {
        try { FrameworkProgram.write(null, Integer.MIN_VALUE); fail(); }
        catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
    }
    @Test public void nested() {
        try { FrameworkProgram.nested(null); fail(); }
        catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
    }
    @Test public void stateAfter() {
        FrameworkProgram p = new FrameworkProgram();
        p.value = Integer.MAX_VALUE; FrameworkProgram.state = Integer.MIN_VALUE;
        try { FrameworkProgram.write(null, 0); fail(); }
        catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
        assertEquals(Integer.MAX_VALUE, p.value);
        assertEquals(Integer.MIN_VALUE, FrameworkProgram.state);
    }
    @Test public void repeated() {
        for (int i = 0; i < 3; i++) {
            try { FrameworkProgram.read(null); fail(); }
            catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
        }
    }
    @Test public void noException() { FrameworkProgram.read(new FrameworkProgram()); fail("missing exception"); }
    @Test public void unexpected() {
        try { FrameworkProgram.unexpected(); fail(); }
        catch (IllegalArgumentException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
    }
    @Test public void wrongOrigin() {
        try { FrameworkProgram.read(null); fail(); }
        catch (NullPointerException e) {
            try { verifyException("java.lang.String", e); fail(); }
            catch (AssertionError expected) { assertTrue(true); }
        }
    }
    @Test public void callback() {
        FrameworkProgram.state = 0;
        verifyException("frameworkfixture.FrameworkProgram", new FrameworkProgram.CustomNPE());
        assertEquals(91, FrameworkProgram.state);
    }
    @Test public void reflection() throws Exception {
        FrameworkProgram.state = 0;
        FrameworkProgram.class.getMethod("callback").invoke(null);
        assertEquals(91, FrameworkProgram.state);
    }
    @Test public void arrayAfter() {
        try { FrameworkProgram.read(null); fail(); }
        catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkProgram", e); }
        int[] a = new int[2]; a[1] = 7; assertEquals(7, a[1]);
    }
    @Test public void repeatedSameException() {
        try { FrameworkProgram.read(null); fail(); }
        catch (NullPointerException e) {
            verifyException("frameworkfixture.FrameworkProgram", e);
            verifyException("frameworkfixture.FrameworkProgram", e);
        }
    }
    @Test public void nullInvoke() {
        FrameworkProgram p = null;
        try { p.toString(); fail(); }
        catch (NullPointerException e) { verifyException("frameworkfixture.FrameworkFixtureTest", e); }
    }
}
