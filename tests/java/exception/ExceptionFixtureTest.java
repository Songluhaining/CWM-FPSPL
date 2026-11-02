package exceptionfixture;
import org.junit.Test;
import static org.junit.Assert.*;

public class ExceptionFixtureTest {
    @Test public void normalTryCatch() { assertEquals(2, ExceptionProgram.normalTryCatch(1)); }
    @Test public void sameMethodThrow() { assertEquals(7, ExceptionProgram.sameMethodThrow()); }
    @Test public void callerCatch() { assertEquals(9, ExceptionProgram.callerCatch()); }
    @Test public void normalFinally() { assertEquals(5, ExceptionProgram.normalFinally()); }
    @Test public void exceptionalFinally() {
        try { ExceptionProgram.exceptionalFinally(); fail("expected exception"); }
        catch (IllegalArgumentException ex) { assertTrue(true); }
    }
    @Test public void ordinaryStore() { assertEquals(1, ExceptionProgram.ordinaryStore()); }
    @Test(expected=NumberFormatException.class) public void untracedException() { Integer.parseInt("not-an-integer"); }
    @Test public void implicitDivide() { assertEquals(-1, ExceptionProgram.implicitDivide(0)); }
}
