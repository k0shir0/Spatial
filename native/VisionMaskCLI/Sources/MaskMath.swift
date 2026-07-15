import Foundation

func isLikelyGreen(r: UInt8, g: UInt8, b: UInt8) -> Bool {
    let red = Int(r)
    let green = Int(g)
    let blue = Int(b)
    let strongestOther = max(red, blue)
    return green >= 42
        && green - strongestOther >= 6
        && green * 100 >= red * 106
        && green * 100 >= blue * 104
}

func isLikelySkin(r: UInt8, g: UInt8, b: UInt8) -> Bool {
    let red = Double(r)
    let green = Double(g)
    let blue = Double(b)
    guard red > 48, green > 28, blue > 18 else { return false }
    guard red > green, red > blue, red - green > 5 else { return false }
    guard max(red, green, blue) - min(red, green, blue) > 14 else { return false }

    let cb = 128.0 - 0.168736 * red - 0.331264 * green + 0.5 * blue
    let cr = 128.0 + 0.5 * red - 0.418688 * green - 0.081312 * blue
    return cb >= 76 && cb <= 136 && cr >= 132 && cr <= 184
}

func dilated(_ mask: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
    guard radius > 0, width > 0, height > 0 else { return mask }
    precondition(mask.count == width * height)

    var horizontal = [UInt8](repeating: 0, count: mask.count)
    for y in 0..<height {
        let rowStart = y * width
        var active = 0
        let initialEnd = min(width - 1, radius)
        if initialEnd >= 0 {
            for x in 0...initialEnd where mask[rowStart + x] != 0 { active += 1 }
        }
        for x in 0..<width {
            horizontal[rowStart + x] = active > 0 ? 1 : 0
            let leaving = x - radius
            if leaving >= 0, mask[rowStart + leaving] != 0 { active -= 1 }
            let entering = x + radius + 1
            if entering < width, mask[rowStart + entering] != 0 { active += 1 }
        }
    }

    var result = [UInt8](repeating: 0, count: mask.count)
    for x in 0..<width {
        var active = 0
        let initialEnd = min(height - 1, radius)
        if initialEnd >= 0 {
            for y in 0...initialEnd where horizontal[y * width + x] != 0 { active += 1 }
        }
        for y in 0..<height {
            result[y * width + x] = active > 0 ? 1 : 0
            let leaving = y - radius
            if leaving >= 0, horizontal[leaving * width + x] != 0 { active -= 1 }
            let entering = y + radius + 1
            if entering < height, horizontal[entering * width + x] != 0 { active += 1 }
        }
    }
    return result
}

func eroded(_ mask: [UInt8], width: Int, height: Int, radius: Int) -> [UInt8] {
    guard radius > 0 else { return mask }
    let inverse = mask.map { $0 == 0 ? UInt8(1) : UInt8(0) }
    let expandedBackground = dilated(inverse, width: width, height: height, radius: radius)
    return expandedBackground.map { $0 == 0 ? UInt8(1) : UInt8(0) }
}

func fillSmallHoles(
    _ mask: [UInt8],
    width: Int,
    height: Int,
    maximumHolePixels: Int
) -> [UInt8] {
    guard maximumHolePixels > 0, width > 0, height > 0 else { return mask }
    precondition(mask.count == width * height)

    var result = mask
    var visited = [UInt8](repeating: 0, count: mask.count)
    var queue: [Int] = []
    queue.reserveCapacity(min(mask.count, maximumHolePixels + 1))

    for start in 0..<mask.count where mask[start] == 0 && visited[start] == 0 {
        queue.removeAll(keepingCapacity: true)
        queue.append(start)
        visited[start] = 1
        var cursor = 0
        var touchesBorder = false

        while cursor < queue.count {
            let current = queue[cursor]
            cursor += 1
            let x = current % width
            let y = current / width
            if x == 0 || y == 0 || x == width - 1 || y == height - 1 {
                touchesBorder = true
            }

            if x > 0 {
                let next = current - 1
                if mask[next] == 0 && visited[next] == 0 { visited[next] = 1; queue.append(next) }
            }
            if x + 1 < width {
                let next = current + 1
                if mask[next] == 0 && visited[next] == 0 { visited[next] = 1; queue.append(next) }
            }
            if y > 0 {
                let next = current - width
                if mask[next] == 0 && visited[next] == 0 { visited[next] = 1; queue.append(next) }
            }
            if y + 1 < height {
                let next = current + width
                if mask[next] == 0 && visited[next] == 0 { visited[next] = 1; queue.append(next) }
            }
        }

        if !touchesBorder && queue.count <= maximumHolePixels {
            for index in queue { result[index] = 1 }
        }
    }
    return result
}

func retainBestGreenComponent(
    _ mask: [UInt8],
    rgba: [UInt8],
    width: Int,
    height: Int
) -> [UInt8] {
    precondition(mask.count == width * height)
    precondition(rgba.count == width * height * 4)

    var labels = [Int32](repeating: 0, count: mask.count)
    var queue: [Int] = []
    var nextLabel: Int32 = 1
    var winningLabel: Int32 = 0
    var winningGreenCount = -1
    var winningSize = -1

    for start in 0..<mask.count where mask[start] != 0 && labels[start] == 0 {
        queue.removeAll(keepingCapacity: true)
        queue.append(start)
        labels[start] = nextLabel
        var cursor = 0
        var greenCount = 0

        while cursor < queue.count {
            let current = queue[cursor]
            cursor += 1
            let pixel = current * 4
            if isLikelyGreen(r: rgba[pixel], g: rgba[pixel + 1], b: rgba[pixel + 2]) {
                greenCount += 1
            }
            let x = current % width
            let y = current / width

            if x > 0 {
                let neighbor = current - 1
                if mask[neighbor] != 0 && labels[neighbor] == 0 { labels[neighbor] = nextLabel; queue.append(neighbor) }
            }
            if x + 1 < width {
                let neighbor = current + 1
                if mask[neighbor] != 0 && labels[neighbor] == 0 { labels[neighbor] = nextLabel; queue.append(neighbor) }
            }
            if y > 0 {
                let neighbor = current - width
                if mask[neighbor] != 0 && labels[neighbor] == 0 { labels[neighbor] = nextLabel; queue.append(neighbor) }
            }
            if y + 1 < height {
                let neighbor = current + width
                if mask[neighbor] != 0 && labels[neighbor] == 0 { labels[neighbor] = nextLabel; queue.append(neighbor) }
            }
        }

        if greenCount > winningGreenCount || (greenCount == winningGreenCount && queue.count > winningSize) {
            winningLabel = nextLabel
            winningGreenCount = greenCount
            winningSize = queue.count
        }
        nextLabel += 1
    }

    guard winningLabel != 0, winningGreenCount > 0 else {
        return [UInt8](repeating: 0, count: mask.count)
    }
    return labels.map { $0 == winningLabel ? UInt8(1) : UInt8(0) }
}

func retainComponentNearestSeed(
    _ mask: [UInt8], width: Int, height: Int, normalizedX: Double, normalizedY: Double
) -> [UInt8] {
    precondition(mask.count == width * height)
    let seedX = min(width - 1, max(0, Int(normalizedX * Double(width))))
    let seedY = min(height - 1, max(0, Int(normalizedY * Double(height))))
    var visited = [Bool](repeating: false, count: mask.count)
    var winner: [Int] = []
    var winningDistance = Double.infinity
    var winningSize = 0

    for start in mask.indices where mask[start] != 0 && !visited[start] {
        var queue = [start]
        visited[start] = true
        var cursor = 0
        var minimumDistance = Double.infinity
        while cursor < queue.count {
            let index = queue[cursor]
            cursor += 1
            let x = index % width
            let y = index / width
            minimumDistance = min(minimumDistance, hypot(Double(x - seedX), Double(y - seedY)))
            for offset in [-width - 1, -width, -width + 1, -1, 1, width - 1, width, width + 1] {
                let neighbor = index + offset
                guard neighbor >= 0, neighbor < mask.count, !visited[neighbor], mask[neighbor] != 0 else { continue }
                let nx = neighbor % width
                guard abs(nx - x) <= 1 else { continue }
                visited[neighbor] = true
                queue.append(neighbor)
            }
        }
        if minimumDistance < winningDistance || (minimumDistance == winningDistance && queue.count > winningSize) {
            winner = queue
            winningDistance = minimumDistance
            winningSize = queue.count
        }
    }
    var result = [UInt8](repeating: 0, count: mask.count)
    for index in winner { result[index] = 1 }
    return result
}
