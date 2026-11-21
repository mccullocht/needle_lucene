import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.text.NumberFormat;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.queryparser.flexible.standard.StandardQueryParser;
import org.apache.lucene.queryparser.flexible.standard.config.PointsConfig;
import org.apache.lucene.search.Query;

/**
 * Validates Lucene query strings with the real StandardQueryParser.
 *
 * <p>Reads lines of {@code numeric_fields<TAB>query} from stdin, where numeric_fields is a
 * comma-separated list (possibly empty) of fields configured as double point fields. Writes one
 * line per input: {@code OK<TAB>parsed query} or {@code ERR<TAB>message}.
 *
 * <p>Run: java -cp 'lucene/lib/*' lucene/Validate.java < input.tsv
 */
public class Validate {
  public static void main(String[] args) throws Exception {
    BufferedReader in =
        new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
    PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
    StandardAnalyzer analyzer = new StandardAnalyzer();
    NumberFormat format = NumberFormat.getNumberInstance(Locale.ROOT);
    format.setGroupingUsed(false);
    PointsConfig doubles = new PointsConfig(format, Double.class);
    String line;
    while ((line = in.readLine()) != null) {
      int tab = line.indexOf('\t');
      String fields = tab < 0 ? "" : line.substring(0, tab);
      String query = tab < 0 ? line : line.substring(tab + 1);
      try {
        StandardQueryParser parser = new StandardQueryParser(analyzer);
        Map<String, PointsConfig> points = new HashMap<>();
        for (String f : fields.split(",")) {
          if (!f.isEmpty()) {
            points.put(f, doubles);
          }
        }
        parser.setPointsConfigMap(points);
        Query q = parser.parse(query, "_default");
        out.println("OK\t" + q.toString().replace('\n', ' '));
      } catch (Throwable t) {
        String msg = String.valueOf(t.getMessage()).replace('\n', ' ').replace('\t', ' ');
        out.println("ERR\t" + t.getClass().getSimpleName() + ": " + msg);
      }
    }
  }
}
