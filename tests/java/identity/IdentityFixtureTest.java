package identityfixture;
import org.junit.Test;
import static org.junit.Assert.*;
public class IdentityFixtureTest {
    @Test public void implicitConstructor() { assertNotNull(new Implicit()); }
    @Test public void explicitConstructor() { assertNotNull(new Explicit()); }
    @Test public void repeatedConstruction() {
        Implicit first = new Implicit();
        Implicit second = new Implicit();
        Explicit third = new Explicit();
        assertNotSame(first, second);
        assertNotNull(third);
    }
    @Test public void initializedField() { assertEquals(7, new WithField().value); }
}
