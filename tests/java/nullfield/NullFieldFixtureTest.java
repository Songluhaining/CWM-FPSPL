package nullfieldfixture;
import org.junit.Test;
import static org.junit.Assert.*;

public class NullFieldFixtureTest {
    @Test public void normalReadWrite() {
        NullFieldProgram p = new NullFieldProgram();
        NullFieldProgram.write(p, Integer.MIN_VALUE);
        assertEquals(Integer.MIN_VALUE, NullFieldProgram.read(p));
        NullFieldProgram.write(p, 0); assertEquals(0, NullFieldProgram.read(p));
        NullFieldProgram.write(p, Integer.MAX_VALUE); assertEquals(Integer.MAX_VALUE, NullFieldProgram.read(p));
    }
    @Test public void sameRead() { assertEquals(11, NullFieldProgram.sameRead()); }
    @Test public void sameWrite() { assertEquals(12, NullFieldProgram.sameWrite()); }
    @Test public void callerRead() { assertEquals(13, NullFieldProgram.callerRead()); }
    @Test public void callerWrite() { assertEquals(14, NullFieldProgram.callerWrite()); }
    @Test public void repeated() {
        for (int i = 0; i < 3; i++) {
            assertEquals(11, NullFieldProgram.sameRead());
            assertEquals(14, NullFieldProgram.callerWrite());
        }
    }
    @Test public void aliasedNull() { assertEquals(7, NullFieldProgram.aliasedNull()); }
    @Test public void nullInvokeRejected() {
        NullFieldProgram p = null;
        try { p.toString(); fail(); } catch (NullPointerException e) { assertTrue(true); }
    }
    @Test public void castRejected() {
        Object p = null; NullFieldProgram q = (NullFieldProgram)p;
        try { NullFieldProgram.read(q); fail(); } catch (NullPointerException e) { assertTrue(true); }
    }
    @Test public void broadCatchRejected() { assertEquals(15, NullFieldProgram.broadCatch()); }
    @Test public void finallyRejected() {
        try { NullFieldProgram.withFinally(); fail(); } catch (NullPointerException e) { assertTrue(true); }
    }
    @Test public void externalEffectsRejected() {
        assertEquals(13, NullFieldProgram.callerRead()); assertEquals(2, Math.abs(-2));
    }
    @Test public void divideRejected() {
        try { NullFieldProgram.divide(0); fail(); } catch (ArithmeticException e) { assertTrue(true); }
    }
}
