import java.io.BufferedReader;
import java.io.InputStreamReader;

public class runPython {
    public static void main(String[] args) throws Exception {
        ProcessBuilder processBuilder = new ProcessBuilder("python", "main.py");
        processBuilder.redirectErrorStream(true);

        Process process = processBuilder.start();
//        process.waitFor();
        BufferedReader bfr = new BufferedReader(new InputStreamReader(process.getInputStream()));
        String line;
        while ((line = bfr.readLine()) != null) {
            System.out.println(line);
        }
    }
}
