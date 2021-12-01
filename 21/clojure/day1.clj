(defn loop [acc prev]
  (let [line (read-line)]
    (if (nil? line)
      (- acc 1)
      (let [line (Integer/parseInt line)]
        (if (> line prev)
          (recur (+ acc 1) line)
          (recur acc line))
        )
      )
    ))

(println (loop 0 -1))
        
