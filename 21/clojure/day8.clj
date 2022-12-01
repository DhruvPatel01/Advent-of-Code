(require '[clojure.string :as str])

(def inp (map #(str/split % #" \| ")
              (.split (slurp "../input/day8") "\n")))

(def part1
  (frequencies
   (map count
        (apply concat (map #(str/split (% 1) #" ")
                           inp)))))

(println (reduce + (map #(part1 %) [2 3 4 7])))
